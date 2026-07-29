"""Slicing invariants: a schedule partitions a document, or it drops tokens."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core._builders import FakeRng
from ttt.core import schedule


@given(
    doc_length=st.integers(min_value=1, max_value=10_000),
    n_slices=st.integers(min_value=1, max_value=32),
)
def test_equal_slices_partition_the_document(doc_length, n_slices):
    spans = schedule.equal_token_slices(doc_length, n_slices)

    assert schedule.covers(spans, doc_length)


@given(
    doc_length=st.integers(min_value=1, max_value=10_000),
    n_slices=st.integers(min_value=1, max_value=32),
)
def test_equal_slices_are_never_empty(doc_length, n_slices):
    spans = schedule.equal_token_slices(doc_length, n_slices)

    assert all(end > start for start, end in spans)


@given(
    doc_length=st.integers(min_value=64, max_value=10_000),
    n_slices=st.integers(min_value=1, max_value=16),
)
def test_equal_slices_differ_in_length_by_at_most_one_token(doc_length, n_slices):
    spans = schedule.equal_token_slices(doc_length, n_slices)

    lengths = [end - start for start, end in spans]
    assert max(lengths) - min(lengths) <= 1


@given(
    doc_length=st.integers(min_value=1, max_value=1000),
    n_slices=st.integers(min_value=1, max_value=64),
)
def test_equal_slices_never_exceed_the_requested_count(doc_length, n_slices):
    spans = schedule.equal_token_slices(doc_length, n_slices)

    assert len(spans) <= n_slices


def test_equal_slices_are_deterministic():
    assert schedule.equal_token_slices(1000, 3) == schedule.equal_token_slices(1000, 3)


def test_equal_slices_rejects_a_zero_slice_count():
    with pytest.raises(ValueError, match="n_slices must be >= 1"):
        schedule.equal_token_slices(100, 0)


@given(
    doc_length=st.integers(min_value=1, max_value=20_000),
    k=st.integers(min_value=1, max_value=10),
    min_slice_tokens=st.integers(min_value=1, max_value=2000),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=60, deadline=None)
def test_random_slices_partition_the_document(doc_length, k, min_slice_tokens, seed):
    spans = schedule.slice_doc(doc_length, k, min_slice_tokens, FakeRng(seed))

    assert schedule.covers(spans, doc_length)


@given(
    doc_length=st.integers(min_value=1, max_value=20_000),
    k=st.integers(min_value=2, max_value=10),
    min_slice_tokens=st.integers(min_value=1, max_value=2000),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=60, deadline=None)
def test_every_random_slice_meets_the_minimum_when_k_is_feasible(
    doc_length, k, min_slice_tokens, seed
):
    spans = schedule.slice_doc(doc_length, k, min_slice_tokens, FakeRng(seed))

    lengths = [end - start for start, end in spans]
    assert min(lengths) >= min(min_slice_tokens, doc_length)


@given(
    doc_length=st.integers(min_value=1, max_value=20_000),
    k=st.integers(min_value=1, max_value=10),
    min_slice_tokens=st.integers(min_value=1, max_value=2000),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=60, deadline=None)
def test_random_slicing_yields_k_slices_or_falls_back_to_one(
    doc_length, k, min_slice_tokens, seed
):
    spans = schedule.slice_doc(doc_length, k, min_slice_tokens, FakeRng(seed))

    assert len(spans) in (1, k)


def test_random_slicing_falls_back_to_one_slice_when_the_minimum_cannot_be_met():
    spans = schedule.slice_doc(1000, k=4, min_slice_tokens=400, rng=FakeRng(0))

    assert spans == ((0, 1000),)


def test_random_slicing_is_deterministic_for_a_seed():
    first = schedule.slice_doc(9000, 4, 1000, FakeRng(7))
    second = schedule.slice_doc(9000, 4, 1000, FakeRng(7))

    assert first == second


def test_random_slicing_varies_with_the_seed():
    first = schedule.slice_doc(9000, 4, 1000, FakeRng(7))
    second = schedule.slice_doc(9000, 4, 1000, FakeRng(8))

    assert first != second


@given(
    doc_length=st.integers(min_value=0, max_value=100_000),
    slice_min_tokens=st.integers(min_value=1, max_value=4000),
    slices_min=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=0, max_value=8),
)
def test_slice_count_stays_within_the_configured_band_or_collapses_to_one(
    doc_length, slice_min_tokens, slices_min, extra
):
    slices_max = slices_min + extra

    k = schedule.derive_slice_count(
        doc_length, slice_min_tokens, slices_min, slices_max
    )

    assert k == 1 or slices_min <= k <= slices_max


@given(
    doc_length=st.integers(min_value=1, max_value=100_000),
    slice_min_tokens=st.integers(min_value=1, max_value=4000),
    slices_min=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=0, max_value=8),
)
def test_a_slice_count_above_one_is_always_feasible(
    doc_length, slice_min_tokens, slices_min, extra
):
    """k > 1 must imply the document can actually carry k minimum-size slices."""
    k = schedule.derive_slice_count(
        doc_length, slice_min_tokens, slices_min, slices_min + extra
    )

    assert k == 1 or k * slice_min_tokens <= doc_length


def test_a_short_document_gets_a_single_slice():
    assert schedule.derive_slice_count(1500, 1000, 2, 6) == 1


def test_a_long_document_is_capped_at_slices_max():
    assert schedule.derive_slice_count(100_000, 1000, 2, 6) == 6


def test_slice_count_rejects_an_inverted_band():
    with pytest.raises(ValueError, match="slices_min <= slices_max"):
        schedule.derive_slice_count(10_000, 1000, 6, 2)


def test_covers_rejects_a_gap():
    assert not schedule.covers(((0, 10), (20, 30)), 30)


def test_covers_rejects_an_overlap():
    assert not schedule.covers(((0, 20), (10, 30)), 30)


def test_covers_rejects_a_short_partition():
    assert not schedule.covers(((0, 10),), 30)

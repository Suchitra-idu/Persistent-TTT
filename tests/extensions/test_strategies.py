"""Behaviour specific to each strategy, beyond the shared contract."""

from __future__ import annotations

import pytest

from ttt.extensions.strategies import (
    EVERLASTING,
    HYBRID,
    MINILASTING,
    SESSION,
    SOURCE,
    Hybrid,
    Minilasting,
)
from tests.extensions import _builders


def test_hybrid_carries_within_a_session_and_everlasting_across_sources():
    assert (HYBRID.carry_scope, EVERLASTING.carry_scope) == (SESSION, SOURCE)


def test_a_doc_below_the_carry_threshold_is_one_whole_item():
    assert HYBRID.slices_for(HYBRID.carry_min_tokens - 1) == 1


def test_a_doc_at_the_carry_threshold_is_sliced():
    assert HYBRID.slices_for(HYBRID.carry_min_tokens) >= HYBRID.slices_min


def test_slice_count_is_capped_at_slices_max():
    assert HYBRID.slices_for(10_000_000) == HYBRID.slices_max


def test_a_zero_length_doc_is_an_error():
    with pytest.raises(ValueError, match="doc length must be >= 1"):
        HYBRID.slices_for(0)


def test_a_carry_threshold_below_the_minimum_slicing_is_an_error():
    with pytest.raises(ValueError, match="carry_min_tokens"):
        Hybrid(carry_min_tokens=500, slice_min_tokens=1000, slices_min=2)


def test_every_hybrid_slice_meets_the_minimum_token_count():
    lengths = [HYBRID.carry_min_tokens * 4] * 20
    sources = _builders.sources_for(lengths, seed=9)

    sessions = HYBRID.build(lengths, sources, _builders.FakeRng(5))

    assert all(
        item.n_tokens >= HYBRID.slice_min_tokens
        for session in sessions
        for item in session.items
    )


def test_hybrid_puts_exactly_one_doc_in_each_session():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    sessions = HYBRID.build(lengths, sources, _builders.FakeRng(5))

    assert all(len(set(session.doc_indices)) == 1 for session in sessions)


def test_everlasting_sessions_are_a_single_whole_doc():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    sessions = EVERLASTING.build(lengths, sources, _builders.FakeRng(5))

    assert all(
        len(session) == 1 and session.items[0].n_tokens == lengths[session.items[0].doc_idx]
        for session in sessions
    )


def test_everlasting_shuffles_the_document_order():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    sessions = EVERLASTING.build(lengths, sources, _builders.FakeRng(5))

    assert [s.items[0].doc_idx for s in sessions] != list(range(len(lengths)))


def test_everlasting_counts_one_item_per_doc():
    lengths = _builders.doc_lengths(30, seed=8)

    assert EVERLASTING.count(lengths) == len(lengths)


def test_everlasting_marks_every_doc_as_carrying():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    rows = EVERLASTING.compose(lengths, sources)

    assert all(row.no_carry_docs == 0 for row in rows)


def test_hybrid_splits_carrying_from_non_carrying_docs():
    lengths = [500, 500, HYBRID.carry_min_tokens * 3]
    sources = ["a", "a", "a"]

    rows = HYBRID.compose(lengths, sources)

    assert (rows[0].no_carry_docs, rows[0].carry_docs) == (2, 1)


def test_minilasting_carries_within_a_session_not_across_it():
    assert MINILASTING.carry_scope == SESSION


def test_minilasting_never_puts_two_sources_in_one_session():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    sessions = Minilasting(docs_per_session=3).build(lengths, sources, _builders.FakeRng(5))

    by_doc = {i: source for i, source in enumerate(sources)}
    assert all(
        len({by_doc[doc_idx] for doc_idx in session.doc_indices}) == 1
        for session in sessions
    )


def test_minilasting_caps_a_session_at_docs_per_session_documents():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    sessions = Minilasting(docs_per_session=3).build(lengths, sources, _builders.FakeRng(5))

    assert all(len(set(session.doc_indices)) <= 3 for session in sessions)


def test_a_wider_session_lets_more_documents_share_a_carry():
    lengths = _builders.doc_lengths(30, seed=8)
    sources = _builders.sources_for(lengths, seed=9)

    narrow = Minilasting(docs_per_session=1).build(lengths, sources, _builders.FakeRng(5))
    wide = Minilasting(docs_per_session=30).build(lengths, sources, _builders.FakeRng(5))

    assert max(len(set(s.doc_indices)) for s in narrow) == 1
    assert max(len(set(s.doc_indices)) for s in wide) > 1


def test_a_docs_per_session_below_one_is_an_error():
    with pytest.raises(ValueError, match="docs_per_session"):
        Minilasting(docs_per_session=0)

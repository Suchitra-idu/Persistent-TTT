"""The one chars-per-token heuristic, and the direction each policy errs in."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ttt.core import tokens


@given(
    n_chars=st.integers(min_value=0, max_value=10_000_000),
    ratio=st.floats(min_value=1.0, max_value=10.0),
)
def test_estimating_tokens_is_monotone_in_length(n_chars, ratio):
    shorter = tokens.estimate_tokens(n_chars, ratio)
    longer = tokens.estimate_tokens(n_chars + 1000, ratio)

    assert longer >= shorter


@given(min_tokens=st.integers(min_value=0, max_value=100_000))
def test_the_char_threshold_round_trips_to_at_least_the_token_minimum(min_tokens):
    """A document exactly at the threshold must not be dropped by the filter."""
    threshold = tokens.min_chars_for(min_tokens)

    assert tokens.estimate_tokens(threshold) >= min_tokens - 1


def test_an_empty_document_has_no_tokens():
    assert tokens.estimate_tokens(0) == 0



"""The one chars-per-token heuristic, and the direction each policy errs in."""

from __future__ import annotations

import pytest
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


@given(
    n_chars=st.integers(min_value=0, max_value=1_000_000),
    min_tokens=st.integers(min_value=0, max_value=10_000),
)
def test_the_filter_agrees_with_the_threshold_it_is_built_from(n_chars, min_tokens):
    assert tokens.has_enough_tokens(n_chars, min_tokens) == (
        n_chars >= tokens.min_chars_for(min_tokens)
    )


def test_the_prefilter_policy_is_permissive():
    """3.5 over-estimates tokens, so borderline docs survive to exact counting."""
    borderline = tokens.min_chars_for(2048, tokens.PREFILTER_CHARS_PER_TOKEN)

    assert borderline < tokens.min_chars_for(2048, tokens.HOLDOUT_CHARS_PER_TOKEN)


def test_the_holdout_policy_is_conservative():
    """4.0 under-estimates tokens; there is no second pass to catch a mistake."""
    chars = 8000

    assert tokens.estimate_tokens(chars, tokens.HOLDOUT_CHARS_PER_TOKEN) < (
        tokens.estimate_tokens(chars, tokens.PREFILTER_CHARS_PER_TOKEN)
    )


def test_an_empty_document_has_no_tokens():
    assert tokens.estimate_tokens(0) == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: tokens.estimate_tokens(-1), "n_chars must be >= 0"),
        (lambda: tokens.estimate_tokens(10, 0.0), "chars_per_token must be > 0"),
        (lambda: tokens.min_chars_for(-1), "min_tokens must be >= 0"),
    ],
)
def test_impossible_arguments_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()

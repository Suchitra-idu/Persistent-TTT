"""The one chars-per-token heuristic (D9).

Prefilter uses 3.5 (permissive: the exact drop_short pass runs after it);
holdout uses 4.0 (conservative: nothing catches a bad pick later).
"""

from __future__ import annotations

PREFILTER_CHARS_PER_TOKEN = 3.5
HOLDOUT_CHARS_PER_TOKEN = 4.0


def estimate_tokens(
    n_chars: int, chars_per_token: float = PREFILTER_CHARS_PER_TOKEN
) -> int:
    """Takes a length, not the text: callers must not materialise millions of rows."""
    if n_chars < 0:
        raise ValueError(f"n_chars must be >= 0, got {n_chars}")
    if chars_per_token <= 0.0:
        raise ValueError(f"chars_per_token must be > 0, got {chars_per_token}")
    return int(n_chars / chars_per_token)


def min_chars_for(
    min_tokens: int, chars_per_token: float = PREFILTER_CHARS_PER_TOKEN
) -> int:
    if min_tokens < 0:
        raise ValueError(f"min_tokens must be >= 0, got {min_tokens}")
    if chars_per_token <= 0.0:
        raise ValueError(f"chars_per_token must be > 0, got {chars_per_token}")
    return int(min_tokens * chars_per_token)


def has_enough_tokens(
    n_chars: int,
    min_tokens: int,
    chars_per_token: float = PREFILTER_CHARS_PER_TOKEN,
) -> bool:
    return n_chars >= min_chars_for(min_tokens, chars_per_token)

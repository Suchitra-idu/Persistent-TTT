"""Partitioning a document into the ranges a session works on.

Primitives only; which documents get sliced is a strategy plugin's call (D4).
Every function returns a contiguous, gapless partition of [0, doc_length).
"""

from __future__ import annotations


def equal_token_slices(doc_length: int, n_slices: int) -> tuple[tuple[int, int], ...]:
    """Reproducible eval slicing. Emits fewer ranges than asked rather than
    empty ones, since a zero-token slice has no perplexity."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    if doc_length < 0:
        raise ValueError(f"doc_length must be >= 0, got {doc_length}")
    boundaries = [round(i * doc_length / n_slices) for i in range(n_slices + 1)]
    boundaries[-1] = doc_length
    return tuple(
        (boundaries[i], boundaries[i + 1])
        for i in range(n_slices)
        if boundaries[i + 1] > boundaries[i]
    )


def slice_doc(
    doc_length: int, k: int, min_slice_tokens: int, rng
) -> tuple[tuple[int, int], ...]:
    """k random contiguous slices, each >= min_slice_tokens.

    Cuts are drawn from the surplus above the mandatory minimum, so every
    slice meets it by arithmetic rather than by rejection. Falls back to one
    whole-document slice when k slices cannot all meet it.
    """
    if doc_length < 0:
        raise ValueError(f"doc_length must be >= 0, got {doc_length}")
    if min_slice_tokens < 0:
        raise ValueError(f"min_slice_tokens must be >= 0, got {min_slice_tokens}")
    if k <= 1 or k * min_slice_tokens > doc_length:
        return ((0, doc_length),)
    free = doc_length - k * min_slice_tokens
    cuts = sorted(int(rng.integers(0, free + 1)) for _ in range(k - 1))
    boundaries = (
        [0]
        + [c + (i + 1) * min_slice_tokens for i, c in enumerate(cuts)]
        + [doc_length]
    )
    return tuple((boundaries[i], boundaries[i + 1]) for i in range(k))


def derive_slice_count(
    doc_length: int, slice_min_tokens: int, slices_min: int, slices_max: int
) -> int:
    """Slices a document supports, clamped to [slices_min, slices_max].

    Returns 1 when it cannot support slices_min at the minimum size; forcing
    it through the multi-slice path would silently drop tokens.
    """
    if slices_min < 1 or slices_max < slices_min:
        raise ValueError(
            f"require 1 <= slices_min <= slices_max, got ({slices_min}, {slices_max})"
        )
    if slice_min_tokens < 1:
        raise ValueError(f"slice_min_tokens must be >= 1, got {slice_min_tokens}")
    if doc_length < slice_min_tokens * slices_min:
        return 1
    return max(min(slices_max, doc_length // slice_min_tokens), slices_min)


def covers(spans: tuple[tuple[int, int], ...], doc_length: int) -> bool:
    """Exported so strategy contract suites assert partitions the same way."""
    if not spans:
        return doc_length == 0
    if spans[0][0] != 0 or spans[-1][1] != doc_length:
        return False
    return all(earlier[1] == later[0] for earlier, later in zip(spans, spans[1:]))

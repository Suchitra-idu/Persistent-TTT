"""Shared plumbing for the RULER task generators: not a task itself, hence
the underscore (matches `_registry.py`)."""

from __future__ import annotations

from typing import Callable, Sequence


GROW = 1.5
MAX_ITERS = 40


def choice(rng, items) -> str:
    return items[rng.integers(0, len(items))]


def sample(rng, items: Sequence[str], k: int) -> list[str]:
    """`k` distinct items, order from `rng.permutation` (no replacement)."""
    return [items[i] for i in sample_indices(rng, len(items), k)]


def sample_indices(rng, n: int, k: int) -> list[int]:
    """`k` distinct indices into `range(n)`, no replacement."""
    if k > n:
        raise ValueError(f"cannot sample {k} distinct indices from {n}")
    return rng.permutation(n)[:k]


def fit_size(
    make: Callable[[int], tuple[str, tuple[str, ...]]],
    *,
    tokenizer,
    seq_len: int,
    reserve: int,
    start: int = 50,
) -> tuple[str, tuple[str, ...]]:
    """Grows `size` (passed to `make`) to roughly fill `seq_len - reserve`
    tokens, then backs off to the last size that fit. A simpler stand-in for
    NVIDIA/RULER's binary search — same goal, one direction at a time."""
    size = max(1, start)
    best: tuple[str, tuple[str, ...]] | None = None
    for _ in range(MAX_ITERS):
        try:
            text, targets = make(size)
        except ValueError:
            # Too small even to lay out the fixed content (e.g. not enough
            # chunks to bury the needles in) — treat like "too big", shrink.
            if size <= 1:
                raise ValueError(f"could not fit an example within {seq_len} tokens") from None
            size = max(1, size // 2)
            continue
        if len(tokenizer.encode(text)) + reserve <= seq_len:
            best = (text, targets)
            size = max(size + 1, int(size * GROW))
            continue
        if best is not None:
            return best
        if size <= 1:
            raise ValueError(f"cannot fit a single-unit example within {seq_len} tokens")
        size = max(1, size // 2)
    if best is None:
        raise ValueError(f"could not fit an example within {seq_len} tokens")
    return best


def interleave(rng, chunks: Sequence[str], items) -> str:
    """`items` inserted at distinct random depths among `chunks`, then joined.

    Shared by niah (needle sentences) and vt (assignment lines) — both bury
    `len(items)` markers inside `len(chunks)` filler pieces at random positions.
    """
    depths = sorted(sample_indices(rng, len(chunks) + 1, len(items)))
    pieces: list[str] = []
    cursor = 0
    for depth, item in zip(depths, items):
        pieces.append(" ".join(chunks[cursor:depth]))
        pieces.append(item)
        cursor = depth
    pieces.append(" ".join(chunks[cursor:]))
    return " ".join(p for p in pieces if p)


def interleave_groups(rng, chunks: Sequence[str], groups: Sequence[Sequence[str]]) -> str:
    """Like `interleave`, but each group's own item order survives the
    shuffle — only which group goes next is random (a riffle shuffle).

    vt's chains need this: "VAR C = VAR B" is only answerable if "VAR B = ..."
    was seen first, so a chain's own steps can never be reordered relative to
    each other, even though which chain comes first is arbitrary.
    """
    return interleave(rng, chunks, _riffled(rng, groups))


def _riffled(rng, groups: Sequence[Sequence[str]]) -> list[str]:
    queues = [list(g) for g in groups if g]
    merged: list[str] = []
    while queues:
        pick = rng.integers(0, len(queues))
        merged.append(queues[pick].pop(0))
        if not queues[pick]:
            queues.pop(pick)
    return merged


def repeated(items: Sequence[str], n: int) -> list[str]:
    """`items`, repeated as needed to reach length `n` (RULER pads a haystack
    shorter than the requested size rather than erroring)."""
    if not items:
        raise ValueError("need at least one item to repeat")
    if n <= len(items):
        return list(items[:n])
    repeats = -(-n // len(items))
    return (list(items) * repeats)[:n]

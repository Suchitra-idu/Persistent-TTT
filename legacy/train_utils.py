"""Pure training utilities extracted from train_modal.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionItem:
    """One unit of work in a session: a contiguous token range of a doc."""

    doc_idx: int
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


def make_session_schedule(num_docs: int, lo: int, hi: int, rng) -> list:
    """Shuffle all docs, partition the order into sessions of size n ~ Uniform[lo, hi]."""
    order = rng.permutation(num_docs)
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(lo, hi + 1))
        sessions.append(order[i:i + n].tolist())
        i += n
    return sessions


def slice_doc(doc_length: int, k: int, min_slice_tokens: int, rng) -> list:
    """Partition [0, doc_length] into k contiguous slices each >= min_slice_tokens."""
    if k <= 1 or k * min_slice_tokens > doc_length:
        return [(0, doc_length)]
    free = doc_length - k * min_slice_tokens
    cuts = sorted(int(rng.integers(0, free + 1)) for _ in range(k - 1))
    boundaries = (
        [0]
        + [c + (i + 1) * min_slice_tokens for i, c in enumerate(cuts)]
        + [doc_length]
    )
    return [(boundaries[i], boundaries[i + 1]) for i in range(k)]


def equal_token_slices(doc_length: int, n_slices: int) -> list:
    """Partition [0, doc_length] into n_slices consecutive ranges of roughly equal size."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    boundaries = [round(i * doc_length / n_slices) for i in range(n_slices + 1)]
    boundaries[-1] = doc_length
    return [(boundaries[i], boundaries[i + 1])
            for i in range(n_slices)
            if boundaries[i + 1] > boundaries[i]]


def make_slice_sessions(
    num_docs: int,
    doc_lengths,
    rng,
    *,
    session_papers: tuple,
    slice_prob: float,
    slice_range: tuple,
    min_slice_tokens: int,
    shuffle: bool = True,
) -> list:
    """Build sessions of SessionItems.

    Session size drawn Uniform(session_papers[0], session_papers[1]).
    Each paper is sliced into k ~ Uniform(slice_range) with probability slice_prob.
    k decrements toward feasibility so a doc that could take k=3 slices but not k=4
    yields 3 items, not silently 1.

    shuffle=False keeps doc order intact (used by inference where the caller
    controls order); training shuffles between epochs to decorrelate.

    Single-paper-per-session mode: session_papers=(1, 1), slice_prob=1.0,
    slice_range=(min, max).
    Multi-paper mode: session_papers=(lo, hi), slice_prob=p, slice_range=(min, max).
    """
    order = (rng.permutation(num_docs).tolist() if shuffle
             else list(range(num_docs)))
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(session_papers[0], session_papers[1] + 1))
        session = []
        for doc_idx in order[i:i + n]:
            L = int(doc_lengths[doc_idx])
            if slice_range[1] > 1 and rng.random() < slice_prob:
                k = int(rng.integers(slice_range[0], slice_range[1] + 1))
                while k > 1 and k * min_slice_tokens > L:
                    k -= 1
            else:
                k = 1
            for s, e in slice_doc(L, k, min_slice_tokens, rng):
                session.append(SessionItem(int(doc_idx), s, e))
        sessions.append(session)
        i += n
    return sessions


def expected_items_per_doc(slice_prob: float, slice_min: int,
                           slice_max: int) -> float:
    """Coarse expectation of SessionItems per paper, used to size the LR schedule."""
    if slice_max <= 1 or slice_prob <= 0:
        return 1.0
    return (1.0 - slice_prob) + slice_prob * 0.5 * (slice_min + slice_max)


def derive_slice_count(doc_length: int, slice_min_tokens: int,
                       slices_min: int, slices_max: int) -> int:
    """How many slices a doc gets in hybrid mode.

    Clamped to [slices_min, slices_max] under the constraint that every
    slice has at least `slice_min_tokens`. Returns 1 for docs that can't
    support even `slices_min` slices at the min-tokens threshold, since
    forcing the k*n bound below the doc length would silently drop tokens.
    """
    if slices_min < 1 or slices_max < slices_min:
        raise ValueError(
            f"require 1 <= slices_min <= slices_max, "
            f"got ({slices_min}, {slices_max})"
        )
    if slice_min_tokens < 1:
        raise ValueError(
            f"slice_min_tokens must be >= 1, got {slice_min_tokens}"
        )
    if doc_length < slice_min_tokens * slices_min:
        return 1
    k = min(slices_max, doc_length // slice_min_tokens)
    return max(k, slices_min)


def make_hybrid_sessions(
    num_docs: int,
    doc_lengths,
    rng,
    *,
    carry_min_tokens: int,
    slice_min_tokens: int,
    slices_min: int,
    slices_max: int,
    shuffle: bool = True,
) -> list:
    """One-doc-per-session with a length-dependent split:

    - `L < carry_min_tokens`: a single-item session (whole doc, no slice).
      Session-training still calls `reset_session_state` at session start,
      so the model sees the "S_0 = 0" case for these docs.
    - `L >= carry_min_tokens`: a single-paper session sliced into k in
      [slices_min, slices_max] pieces, each with >= slice_min_tokens
      tokens. Carry propagates across the k slices via TBPTT.

    Intended for pretraining mixes (SlimPajama et al.) where document
    length varies wildly; the point of training with carry is to teach the
    model to use a non-zero fast-weight initialization when it exists,
    which is fine to skip on the short-doc tail.
    """
    if carry_min_tokens < slices_min * slice_min_tokens:
        # A doc at the boundary would be forced through the multi-slice
        # path but couldn't meet the minimum. Fail loudly instead of
        # silently collapsing it back to k=1.
        raise ValueError(
            f"carry_min_tokens={carry_min_tokens} < slices_min * "
            f"slice_min_tokens = {slices_min * slice_min_tokens}; "
            f"raise carry_min_tokens or lower slices_min / slice_min_tokens"
        )
    order = (rng.permutation(num_docs).tolist() if shuffle
             else list(range(num_docs)))
    sessions = []
    for doc_idx in order:
        L = int(doc_lengths[doc_idx])
        if L < carry_min_tokens:
            sessions.append([SessionItem(int(doc_idx), 0, L)])
            continue
        k = derive_slice_count(L, slice_min_tokens, slices_min, slices_max)
        sessions.append([
            SessionItem(int(doc_idx), s, e)
            for s, e in slice_doc(L, k, slice_min_tokens, rng)
        ])
    return sessions


def total_hybrid_items(doc_lengths, *, carry_min_tokens: int,
                       slice_min_tokens: int, slices_min: int,
                       slices_max: int) -> int:
    """Sum of SessionItems produced by `make_hybrid_sessions` over these
    doc lengths; used by the training loop to size the LR schedule."""
    total = 0
    for L in doc_lengths:
        L = int(L)
        if L < carry_min_tokens:
            total += 1
        else:
            total += derive_slice_count(
                L, slice_min_tokens, slices_min, slices_max,
            )
    return total
